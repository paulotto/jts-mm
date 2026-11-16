# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Hub memory stays bounded when nobody drains it (recording without a viewer)."""

from __future__ import annotations

import numpy as np

from jtsmm.viewer.hub import IMU_MOTION_KEEP_S, IMU_MOTION_MAX, Hub
from jtsmm.viewer.protocol import IMU_MOTION_STREAMS


def test_motion_buffers_are_bounded_without_snapshots():
    hub = Hub(emg_fs=2000.0, emg_lsb_uv=0.1, n_channels=7, filter_enabled=False)
    hub.append_emg_raw(np.array([0], dtype=np.uint64), np.zeros((1, 7), np.float32))
    # 10 minutes of 400 Hz on every stream of both slots, never snapshotted.
    n = 400 * 600
    t = (np.arange(n) * 2500).astype(np.uint64)
    v = np.zeros((n, 3), dtype=np.float32)
    for slot in ("mta", "cra"):
        for stream in IMU_MOTION_STREAMS:
            hub.push_imu_motion_series(slot, stream, t, v)
    for slot in ("mta", "cra"):
        for stream in IMU_MOTION_STREAMS:
            dq = hub._imu_motion[slot][stream]
            assert len(dq) <= min(IMU_MOTION_MAX, int(IMU_MOTION_KEEP_S * 400) + 1)
    snap = hub.snapshot()
    ts, _ = snap.imu_mta.motion["accel"]
    assert ts[-1] - ts[0] <= IMU_MOTION_KEEP_S + 1e-3
    assert not hub._imu_motion["mta"]["accel"]           # drained


def test_motion_buffers_bounded_before_emg_sets_t0():
    hub = Hub(emg_fs=2000.0, emg_lsb_uv=0.1, n_channels=7, filter_enabled=False)
    for i in range(3 * IMU_MOTION_MAX):
        hub.push_imu_motion("mta", "gyro", 1000 + i, (0.0, 0.0, 0.0))
    assert len(hub._imu_motion["mta"]["gyro"]) <= IMU_MOTION_MAX
