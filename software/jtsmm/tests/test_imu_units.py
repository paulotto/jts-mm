# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""BNO085 int16 Q-point -> SI conversion golden vectors."""

from __future__ import annotations

import numpy as np
import pytest

from jtsmm.imu import units


def test_accel_q8():
    # 256 counts = 1 g worth of Q8 == 1.0 m/s^2 per count/256.
    assert units.accel_to_si((256, -512, 128)) == pytest.approx((1.0, -2.0, 0.5))


def test_gyro_q9():
    assert units.gyro_to_si((512, -256, 0)) == pytest.approx((1.0, -0.5, 0.0))


def test_mag_q4():
    assert units.mag_to_si((16, 8, -32)) == pytest.approx((1.0, 0.5, -2.0))


def test_raw_to_si_dispatch():
    assert units.raw_to_si(0x01, (256, 0, 0)) == pytest.approx((1.0, 0.0, 0.0))
    assert units.raw_to_si(0x02, (512, 0, 0)) == pytest.approx((1.0, 0.0, 0.0))
    assert units.raw_to_si(0x03, (16, 0, 0)) == pytest.approx((1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        units.raw_to_si(0x99, (1, 2, 3))


def test_raw_array_to_si_matches_scalar():
    counts = np.array([[256, 512, -256], [0, 128, 0]], dtype=np.int16)
    out = units.raw_array_to_si(0x01, counts)
    assert out.dtype == np.float32
    assert out[0] == pytest.approx((1.0, 2.0, -1.0))
    assert out[1] == pytest.approx((0.0, 0.5, 0.0))
