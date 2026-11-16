# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""imu_rate_report: gap-excluded rate, gap count, configured fusion rate."""

from __future__ import annotations

import numpy as np
import pytest

from jtsmm.tools.imu_rate_report import gap_stats, report_session


def test_gap_excluded_rate_and_count():
    first = np.arange(4000) * 2500.0  # 400 Hz ...
    t_us = np.r_[first, first[-1] + 10e6 + np.arange(4000) * 2500.0]  # ... then a 10 s hole
    active, n_gaps, gap_s = gap_stats(t_us)
    assert active == pytest.approx(400.0, rel=1e-6)
    assert n_gaps == 1
    assert gap_s == pytest.approx(10.0, abs=0.01)
    overall = (t_us.size - 1) / ((t_us[-1] - t_us[0]) / 1e6)
    assert overall == pytest.approx(7999 / 30.0, rel=1e-3)  # old number: hole counted as time


def test_report_reads_configured_fusion_rate(tmp_path):
    from jtsmm.io.hdf5_writer import Hdf5Writer
    from jtsmm.sync.clock_align import AlignedRecord

    mcu = {
        "schema_version": 1,
        "imu_mta": {
            "fusion": {"primary": "game_rotation_vector", "rate_hz": 390},
            "reports": {"game_rotation_vector": {"enabled": True, "rate_hz": 390},
                        "linear_acceleration": {"enabled": True, "rate_hz": 250}},
        },
    }
    path = tmp_path / "session.h5"
    w = Hdf5Writer(path=path, config_hash="beef", mcu_config=mcu, host_config={})
    w.open()
    t = [int(i * 1e6 / 390) for i in range(390)] + [int(3e6 + i * 1e6 / 390) for i in range(390)]
    for tq in t:  # 1 s, a 2 s hole, 1 s
        w.append(AlignedRecord("imu_mta", tq, {"quat": (1.0, 0.0, 0.0, 0.0), "quat_report_id": 0x08}))
    w.close()

    [r] = [r for r in report_session(path) if r.stream == "quat"]
    assert r.configured_hz == 390
    assert r.n_gaps == 1 and r.gap_s == pytest.approx(2.0, abs=0.01)
    assert r.active_hz == pytest.approx(390, rel=0.01)
    assert r.measured_hz < 200
