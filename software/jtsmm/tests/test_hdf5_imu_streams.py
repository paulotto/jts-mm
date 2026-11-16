# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Round-trip every enabled IMU stream through the Python HDF5 writer/reader."""

from __future__ import annotations

import numpy as np
import pytest

from jtsmm.imu import units
from jtsmm.io.hdf5_writer import Hdf5Writer
from jtsmm.io.session_reader import Session
from jtsmm.sync.clock_align import AlignedRecord


def _mcu_config() -> dict:
    reports = {
        "game_rotation_vector": {"enabled": True, "rate_hz": 390},
        "gyroscope": {"enabled": True, "rate_hz": 400},
        "magnetometer": {"enabled": True, "rate_hz": 100},
        "linear_acceleration": {"enabled": True, "rate_hz": 200},
    }
    slot = {"fusion": {"primary": "game_rotation_vector"}, "reports": reports}
    return {"schema_version": 1, "imu_mta": slot, "imu_cra": slot}


def _write_session(path):
    w = Hdf5Writer(path=path, config_hash="deadbeef", mcu_config=_mcu_config(),
                   host_config={})
    w.open()
    # Two quat samples (tagged game_rv 0x08), and one of each motion vector.
    w.append(AlignedRecord("imu_mta", 1000, {"quat": (1.0, 0.0, 0.0, 0.0), "quat_report_id": 0x08}))
    w.append(AlignedRecord("imu_mta", 2000, {"quat": (0.0, 1.0, 0.0, 0.0), "quat_report_id": 0x08}))
    w.append(AlignedRecord("imu_mta", 1100, {"accel": (256, -512, 128), "sensor_id": 0x01}))
    w.append(AlignedRecord("imu_mta", 1200, {"gyro": (512, 0, -256), "sensor_id": 0x02}))
    w.append(AlignedRecord("imu_mta", 1300, {"mag": (16, 8, -32), "sensor_id": 0x03}))
    w.append(AlignedRecord("imu_mta", 1400, {"linear_accel": (0.5, -1.0, 9.8)}))
    w.close()


def test_all_imu_streams_roundtrip(tmp_path):
    path = tmp_path / "session.h5"
    _write_session(path)

    with Session.open(path) as s:
        imu = s.imu_mta
        assert imu is not None
        # Quaternion + tagged report id.
        assert imu.n == 2
        assert imu.quat_report_id is not None
        assert set(int(x) for x in imu.quat_report_id) == {0x08}

        streams = imu.motion_streams()
        assert set(streams) == {"accel", "gyro", "mag", "linear_accel"}

        # SI conversion applied by the writer.
        assert tuple(streams["accel"].data[0]) == pytest.approx(units.accel_to_si((256, -512, 128)))
        assert tuple(streams["gyro"].data[0]) == pytest.approx(units.gyro_to_si((512, 0, -256)))
        assert tuple(streams["mag"].data[0]) == pytest.approx(units.mag_to_si((16, 8, -32)))
        assert tuple(streams["linear_accel"].data[0]) == pytest.approx((0.5, -1.0, 9.8))

        # Per-stream timestamps and unit/rate attrs.
        assert int(streams["accel"].t_us[0]) == 1100
        assert streams["accel"].units == "m/s^2"
        assert streams["gyro"].units == "rad/s"
        assert streams["mag"].units == "uT"
        assert streams["gyro"].rate_hz == 400


def test_legacy_quat_only_session_still_loads(tmp_path):
    path = tmp_path / "legacy.h5"
    w = Hdf5Writer(path=path, config_hash="beef", mcu_config=_mcu_config(), host_config={})
    w.open()
    w.append(AlignedRecord("imu_mta", 1000, {"quat": (1.0, 0.0, 0.0, 0.0), "quat_report_id": None}))
    w.close()
    with Session.open(path) as s:
        assert s.imu_mta is not None
        assert s.imu_mta.n == 1
        assert s.imu_mta.motion_streams() == {}
