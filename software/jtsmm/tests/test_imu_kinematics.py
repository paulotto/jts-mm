# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Derived relative jaw kinematics: q_rel identities, ZUPT drift bounding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jtsmm.imu import kinematics as kin


def test_relative_quaternion_identity_when_equal():
    q = kin._normalize((0.5, 0.5, 0.5, 0.5))
    q_rel = kin.relative_quaternion(q, q)
    # Equal orientations -> identity rotation (w=+/-1, vector ~0).
    assert abs(abs(q_rel[0]) - 1.0) < 1e-6
    assert max(abs(c) for c in q_rel[1:]) < 1e-6


def test_relative_quaternion_composition():
    # cra rotates 90 deg about z, mta rotates 180 deg about z -> rel is 90 about z.
    def rz(deg):
        r = math.radians(deg) / 2
        return (math.cos(r), 0.0, 0.0, math.sin(r))

    q_rel = kin.relative_quaternion(rz(180), rz(90))
    expected = rz(90)
    dot = abs(sum(a * b for a, b in zip(q_rel, expected)))
    assert dot == pytest.approx(1.0, abs=1e-6)


def test_quat_rotate_identity():
    assert kin.quat_rotate((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)) == pytest.approx((1.0, 2.0, 3.0))


def test_quat_rotate_90_about_z():
    r = math.radians(90) / 2
    q = (math.cos(r), 0.0, 0.0, math.sin(r))
    out = kin.quat_rotate(q, (1.0, 0.0, 0.0))
    assert out == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)


def test_relative_linear_accel_cancels_common_motion():
    # Both bodies at identity orientation with identical accel -> zero relative.
    a = (0.1, -0.2, 9.8)
    rel = kin.relative_linear_accel(a, (1, 0, 0, 0), a, (1, 0, 0, 0))
    assert rel == pytest.approx((0.0, 0.0, 0.0))


def test_quat_at_time_interpolates_between_samples():
    samples = [
        (0, (1.0, 0.0, 0.0, 0.0)),
        (1_000_000, (0.0, 0.0, 0.0, 1.0)),  # 90 deg about z
    ]
    q_mid = kin.quat_at_time(samples, 500_000)
    assert q_mid is not None
    # Midpoint slerp between identity and 90-z should have nonzero z component.
    assert abs(q_mid[3]) > 0.3


def test_quat_slerp_endpoints():
    q0 = (1.0, 0.0, 0.0, 0.0)
    q1 = kin._normalize((0.0, 0.0, 0.0, 1.0))
    assert kin.quat_slerp(q0, q1, 0.0) == pytest.approx(q0, abs=1e-6)
    assert kin.quat_slerp(q0, q1, 1.0) == pytest.approx(q1, abs=1e-6)


def test_zupt_zeros_velocity_when_stationary():
    z = kin.ZuptIntegrator(zupt_threshold=0.3, zupt_hold_s=0.05, auto_reset_s=100.0)
    t = 0.0
    dt = 0.01
    # A brief impulse then rest: velocity should decay to zero (ZUPT), bounding drift.
    for i in range(500):
        a = (5.0, 0.0, 0.0) if i < 3 else (0.0, 0.0, 0.0)
        t += dt
        pos, valid = z.update(t, a)
        assert valid
    # After ZUPT, velocity is clamped so position stops growing.
    assert np.linalg.norm(z._v) == pytest.approx(0.0, abs=1e-9)
    assert abs(z.position[0]) < 1.0  # bounded, not runaway


def test_zupt_auto_reset():
    z = kin.ZuptIntegrator(zupt_threshold=0.0, auto_reset_s=0.1)
    t = 0.0
    saw_reset = False
    for _ in range(100):
        t += 0.01
        z.update(t, (1.0, 0.0, 0.0))
        if z.did_reset:
            saw_reset = True
    assert saw_reset
