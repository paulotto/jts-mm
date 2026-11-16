# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Replay: seek is adopted by the replay thread only; speed is validated."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from jtsmm.viewer.sources import ReplaySource, validate_replay_speed
from tests.test_viewer import _write_session


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan, 0.0, -1.0, "fast"])
def test_bad_speeds_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_replay_speed(bad)


def test_speed_is_clamped():
    assert validate_replay_speed(1e9) == 64.0
    assert validate_replay_speed(1e-9) == 0.01


def test_seek_only_bumps_a_generation_and_run_thread_adopts_it(tmp_path: Path):
    p = tmp_path / "s.h5"
    _write_session(p, n_emg=20_000, fs=2000)            # 10 s
    src = ReplaySource(p, speed=1.0)
    hub = src.make_hub()
    idx_before = src._emg_idx
    src.seek(7.5)
    assert src._emg_idx == idx_before                   # not touched by the caller
    assert src.position()["t_s"] == pytest.approx(7.5)
    src.start(hub)
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and src._emg_idx < 15_000:
            time.sleep(0.02)
        assert src._emg_idx >= 15_000                   # continued from 7.5 s
        snap = hub.snapshot()
        if snap.emg_n:
            # No rows from before the seek point after the hub reset.
            assert float(np.min(snap.emg_t)) >= 7.4 - 1e-3 - (hub._t0_us - src._t0) / 1e6
        with pytest.raises(ValueError):
            src.seek(math.nan)
        src.set_speed(2.0)
        with pytest.raises(ValueError):
            src.set_speed(math.inf)
        assert src.position()["speed"] == 2.0
        assert src._thread is not None and src._thread.is_alive()
    finally:
        src.stop()


def test_hub_reset_stream_clears_position_state(tmp_path: Path):
    p = tmp_path / "s.h5"
    _write_session(p, n_emg=4000, fs=2000)
    src = ReplaySource(p)
    hub = src.make_hub()
    t = (np.arange(400) * 500).astype(np.uint64)
    hub.append_emg_raw(t, np.ones((400, 7), np.float32))
    hub.push_imu_motion("mta", "gyro", 1000, (0.0, 0.0, 0.0))
    hub.reset_stream()
    snap = hub.snapshot()
    assert snap.emg_n == 0
    assert not snap.imu_mta.motion
    src.stop()


def test_api_speed_rejects_non_finite_with_400(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from starlette.testclient import TestClient

    from jtsmm.viewer.server import build_app

    p = tmp_path / "s.h5"
    _write_session(p, n_emg=4000, fs=2000)
    src = ReplaySource(p)
    hub = src.make_hub()
    client = TestClient(build_app(src, hub))
    r = client.post("/api/speed", content=b'{"speed": Infinity}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code in (400, 422)
    assert client.post("/api/speed", json={"speed": -2}).status_code == 400
    assert client.post("/api/speed", json={"speed": 4}).json()["speed"] == 4.0
    assert client.post("/api/seek", json={"t_s": "x"}).status_code == 400
    src.stop()
