# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Viewer ZUPT estimate: linear-accel samples paired by timestamp, integrated once."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jtsmm.viewer.hub import Hub

FS = 100.0


def _hub():
    hub = Hub(emg_fs=2000.0, emg_lsb_uv=0.1, n_channels=7, filter_enabled=False)
    hub._zupt.auto_reset_s = 1e9
    return hub


def _series(seconds=1.0, a_mta=(1.0, 0.0, 0.0), a_cra=(0.0, 0.0, 0.0)):
    n = int(seconds * FS) + 1
    t = (np.arange(n) * (1e6 / FS)).astype(np.uint64) + np.uint64(1_000_000)
    return t, np.tile(a_mta, (n, 1)), np.tile(a_cra, (n, 1))


def _quats(hub, t):
    q = np.tile([1.0, 0.0, 0.0, 0.0], (t.size, 1))
    hub.push_imu_quat_series("mta", t, q)
    hub.push_imu_quat_series("cra", t, q)


def test_tap_batches_integrate_each_interval_once():
    """Tap order: whole MTA batch, then whole CRA batch. 1 m/s² for 1 s -> 0.5 m."""
    hub = _hub()
    t, a_m, a_c = _series()
    _quats(hub, t)
    hub.push_imu_motion_series("mta", "linear_accel", t, a_m)
    hub.push_imu_motion_series("cra", "linear_accel", t, a_c)
    pos = hub.snapshot().imu_pos_rel
    assert pos is not None
    assert pos[0] == pytest.approx(0.5, rel=0.03)      # was 1.786 m


def test_live_interleaving_gives_the_same_answer():
    hub = _hub()
    t, a_m, a_c = _series()
    _quats(hub, t)
    for i in range(t.size):
        hub.push_imu_motion("cra", "linear_accel", int(t[i]), tuple(a_c[i]))
        hub.push_imu_motion("mta", "linear_accel", int(t[i]), tuple(a_m[i]))
    assert hub.snapshot().imu_pos_rel[0] == pytest.approx(0.5, rel=0.03)


def test_small_batches_in_either_order():
    hub = _hub()
    t, a_m, a_c = _series()
    _quats(hub, t)
    for s in range(0, t.size, 7):
        hub.push_imu_motion_series("mta", "linear_accel", t[s : s + 7], a_m[s : s + 7])
        hub.push_imu_motion_series("cra", "linear_accel", t[s : s + 7], a_c[s : s + 7])
    assert hub.snapshot().imu_pos_rel[0] == pytest.approx(0.5, rel=0.03)


def test_mounting_rotation_does_not_rotate_chip_frame_accel():
    """Same chip orientation and accel on both slots: relative accel is zero,
    whatever mounting rotation is set for display (a_world = R_wc a_chip)."""
    hub = _hub()
    c, s_ = math.cos(math.radians(60)), math.sin(math.radians(60))
    hub.set_mounting_rotation("mta", np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]))
    t, a_m, _ = _series(a_mta=(2.0, 0.0, 0.0))
    _quats(hub, t)
    hub.push_imu_motion_series("mta", "linear_accel", t, a_m)
    hub.push_imu_motion_series("cra", "linear_accel", t, a_m)
    assert np.allclose(hub.snapshot().imu_pos_rel, 0.0, atol=1e-12)


def test_pending_buffer_is_bounded_when_one_slot_is_silent():
    from jtsmm.viewer.hub import LIN_PAIR_MAX

    hub = _hub()
    t, a_m, _ = _series(seconds=30.0)
    hub.push_imu_motion_series("mta", "linear_accel", t, a_m)
    assert len(hub._lin_pending) <= LIN_PAIR_MAX


def test_gravity_included_accel_feeds_the_estimate():
    """kinematics_accel: no linear_accel, so accel (with +g on chip Z for both
    slots) drives it and gravity cancels in the difference -> still 0.5 m."""
    hub = _hub()
    g = 9.80665
    t, a_m, a_c = _series(a_mta=(1.0, 0.0, g), a_cra=(0.0, 0.0, g))
    _quats(hub, t)
    hub.push_imu_motion_series("mta", "accel", t, a_m)
    hub.push_imu_motion_series("cra", "accel", t, a_c)
    pos = hub.snapshot().imu_pos_rel
    assert pos is not None
    assert pos[0] == pytest.approx(0.5, rel=0.03)
    assert abs(pos[2]) < 1e-6


def test_linear_accel_wins_over_accel_when_both_stream():
    hub = _hub()
    t, a_m, a_c = _series()
    _quats(hub, t)
    junk = np.tile([5.0, 0.0, 9.8], (t.size, 1))
    hub.push_imu_motion_series("mta", "accel", t[:10], junk[:10])
    hub.push_imu_motion_series("mta", "linear_accel", t, a_m)
    hub.push_imu_motion_series("mta", "accel", t, junk)  # ignored from now on
    hub.push_imu_motion_series("cra", "accel", t, junk)
    hub.push_imu_motion_series("cra", "linear_accel", t, a_c)
    pos = hub.snapshot().imu_pos_rel
    assert pos[0] == pytest.approx(0.5, rel=0.03)
