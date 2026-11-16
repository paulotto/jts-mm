# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2025 Paul-Otto Müller
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""ArUco adapter: schema-version check and time-correct frame placement."""

import json
import logging

import h5py
import numpy as np
import pytest

from jts import aruco_camera as acam


def _write_optical(path, t_us, *, schema_version=1, triggers=None, rate_hz=None,
                   nan_rows=()):
    """Multimodal session with one /optical row per ``t_us`` entry.

    MP moves along X at 30 mm/s so position encodes time; ``triggers`` is an
    optional ``(t_master_us, trigger_id)`` pair written to ``/triggers``.
    """
    t_us = np.asarray(t_us, dtype=np.uint64)
    n = t_us.shape[0]
    x = (t_us.astype(np.float64) - float(t_us[0])) * 30e-6
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = schema_version
        opt = f.create_group("optical")
        opt.create_dataset("t_master_us", data=t_us)
        for ds in ("pose_cra", "pose_mta", "pose_ct"):
            poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
            if ds == "pose_mta":
                poses[:, 0, 3] = x
            for i in nan_rows:
                poses[i] = np.nan
            opt.create_dataset(ds, data=poses)
        if triggers is not None:
            tg = f.create_group("triggers")
            tg.create_dataset("t_master_us", data=np.asarray(triggers[0], dtype=np.uint64))
            tg.create_dataset("trigger_id", data=np.asarray(triggers[1], dtype=np.uint64))
            if rate_hz is not None:
                tg.attrs["rate_hz"] = rate_hz


@pytest.mark.parametrize("version", [1, "1", "1.0", np.int64(1)])
def test_schema_version_int_and_string_accepted(tmp_path, caplog, version):
    path = tmp_path / "s.h5"
    _write_optical(path, np.arange(5) * 33_333, schema_version=version)
    with caplog.at_level(logging.WARNING):
        acam.ArucoCameraData(str(path))
    assert "schema version" not in caplog.text


def test_schema_version_other_major_warns(tmp_path, caplog):
    path = tmp_path / "s.h5"
    _write_optical(path, np.arange(5) * 33_333, schema_version="2.0")
    with caplog.at_level(logging.WARNING):
        acam.ArucoCameraData(str(path))
    assert "schema version" in caplog.text


# --------------------------------------------------------------------------- #
# 30 Hz trigger grid with dropped frames
# --------------------------------------------------------------------------- #
PERIOD_US = 33_333
N_TRIG = 60
DROPPED = (5, 6, 7, 20, 41)


def _grid_session(path, *, with_triggers=True, mcu_rate=None):
    trig_t = 10_000_000 + np.arange(N_TRIG, dtype=np.int64) * PERIOD_US
    trig_id = np.arange(100, 100 + N_TRIG)
    keep = np.setdiff1d(np.arange(N_TRIG), DROPPED)
    t_rows = trig_t[keep] + (np.arange(keep.size) % 3 - 1) * 4  # a few µs stamp jitter
    t_rows[0] = trig_t[keep[0]]
    _write_optical(path, t_rows, triggers=(trig_t, trig_id) if with_triggers else None)
    if mcu_rate is not None:
        with h5py.File(path, "a") as f:
            f["meta"].attrs["mcu_config_json"] = json.dumps({"trigger": {"rate_hz": mcu_rate}})
    return trig_t, keep


@pytest.mark.parametrize(
    "kw",
    [dict(), dict(with_triggers=False), dict(with_triggers=False, mcu_rate=30)],
    ids=["triggers", "rows_only", "mcu_config"],
)
def test_dropped_frames_become_nan_on_trigger_grid(tmp_path, kw):
    path = tmp_path / "s.h5"
    trig_t, keep = _grid_session(path, **kw)
    data = acam.ArucoCameraData(str(path))

    assert data.frames == N_TRIG  # 55 rows -> 60 frames
    np.testing.assert_allclose(data.frame_rate, 1e6 / PERIOD_US, rtol=1e-3)
    mp = data.rigid_bodies["MP"].positions
    assert np.isnan(mp[list(DROPPED)]).all()
    assert np.isfinite(mp[keep]).all()
    # Frame k sits at trigger k: x = 30 mm/s * (t_k - t_0).
    np.testing.assert_allclose(mp[keep, 0], (trig_t[keep] - trig_t[0]) * 30e-6, atol=1e-3)
    # Holes carry the nominal trigger time (period from 30 Hz config: +-20 us).
    np.testing.assert_allclose(data.timestamps["t_master_us"], trig_t, atol=25)
    assert data.timestamps["row_present"].sum() == keep.size


def test_frame_intervals_span_real_time(tmp_path):
    from jts.core import FrameInterval

    path = tmp_path / "s.h5"
    _grid_session(path)
    handler = acam.ArucoCameraMotionCaptureData({"type": "aruco_camera"})
    handler.load_data(str(path))
    # Frames 0..29 = 29 trigger periods even though 4 rows are missing.
    T = handler.compute_relative_transform("HP", "MP", FrameInterval(0, 29))
    assert T.shape[0] == 30
    t = handler.frame_times_us(np.arange(30))
    assert (t[-1] - t[0]) == pytest.approx(29 * PERIOD_US, abs=10)
    assert np.isnan(handler.frame_times_us(np.array([-1])))[0]


def test_legacy_rows_without_timestamps_keep_row_layout(tmp_path):
    """No /optical/t_master_us: nothing to place, frames stay row-indexed."""
    path = tmp_path / "s.h5"
    _write_optical(path, np.arange(5) * PERIOD_US)
    with h5py.File(path, "a") as f:
        del f["optical/t_master_us"]
    data = acam.ArucoCameraData(str(path))
    assert data.frames == 5
    assert data.grid_period_us is None


def test_connected_intervals_carry_source_frames(tmp_path):
    from jts.core import FrameInterval, InterpolationMethod, MultiIntervalProcessor

    path = tmp_path / "s.h5"
    trig_t, _keep = _grid_session(path)
    handler = acam.ArucoCameraMotionCaptureData({"type": "aruco_camera"})
    handler.load_data(str(path))
    proc = MultiIntervalProcessor(InterpolationMethod.CUBIC, transition_frames=4)
    ivs = [FrameInterval(8, 15), FrameInterval(30, 39)]
    T = proc.process_intervals(
        lambda iv: handler.compute_relative_transform("HP", "MP", iv), ivs, connect=True
    )
    src = proc.source_frames
    assert T.shape[0] == src.shape[0] == 8 + 4 + 10
    np.testing.assert_array_equal(src[:8], np.arange(8, 16))
    np.testing.assert_array_equal(src[8:12], -1)  # synthetic bridge
    np.testing.assert_array_equal(src[12:], np.arange(30, 40))
    # Real rows keep their pose (no duplicated or dropped endpoint frame).
    first = handler.compute_relative_transform("HP", "MP", FrameInterval(30, 30))[0]
    np.testing.assert_allclose(T[12], first)
    t = handler.frame_times_us(src)
    assert np.isnan(t[8:12]).all()
    np.testing.assert_allclose(t[12:], trig_t[30:40], atol=5)
