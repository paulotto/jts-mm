# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Tracking HDF5 write/read round trip (pure numpy + h5py; no cv2)."""

from __future__ import annotations

import json
import shutil

import h5py
import numpy as np
import pytest

from optical.tracking.io import (
    FrameRecord,
    TrackingSessionWriter,
    read_markers_3d,
    read_tracking_session,
    write_tracking_session,
)
from optical.tracking.pose import nan_pose


def _frame(i: int, *, cams=("cam0", "cam1"), with_markers=True, period_us=33_333) -> FrameRecord:
    fr = FrameRecord(
        frame_idx=i,
        t_cam_ns={c: 1_000_000_000 + i * period_us * 1000 + k for k, c in enumerate(cams)},
        t_master_us=5_000_000 + i * period_us,
    )
    T = np.eye(4)
    T[:3, 3] = (i, 2.0 * i, 600.0)
    fr.set_pose("HP", T)
    fr.set_pose("MP", nan_pose() if i % 3 == 0 else T)
    fr.quality = {
        "HP": {"n_tags": 2, "n_cameras": 2, "reprojection_rms_px": 0.4, "fit_rms_mm": 0.2},
        "MP": {"n_tags": 1, "n_cameras": 1, "reprojection_rms_px": 0.9, "fit_rms_mm": 7.0},
    }
    if with_markers:
        fr.markers_3d = [
            {"marker_id": 1, "corner_index": c, "position_mm": [i, c, 600.0],
             "n_views": 2, "reprojection_rms_px": 0.3}
            for c in range(4)
        ]
    return fr


def test_round_trip_flat_layout(tmp_path):
    frames = [_frame(i) for i in range(10)]
    frames[4].markers_3d = None  # a frame without triangulated corners
    path = write_tracking_session(
        tmp_path / "s.h5",
        frames=frames,
        rig_id="rig",
        method="triangulation",
        extra_meta={"sync_report": {"drops": {"cam0": 1}}, "pair_mismatch": 3},
    )
    d = read_tracking_session(path)
    assert d["meta"]["n_frames"] == 10
    assert d["meta"]["frame_rate_hz"] == pytest.approx(30.0, rel=1e-3)
    assert json.loads(d["meta"]["sync_report"]) == {"drops": {"cam0": 1}}
    assert int(d["meta"]["pair_mismatch"]) == 3
    assert list(d["timestamps"]["frame_idx"]) == list(range(10))
    assert d["bodies"]["HP"]["tracking_valid"].all()
    assert list(d["bodies"]["MP"]["tracking_valid"]) == [i % 3 != 0 for i in range(10)]
    assert np.allclose(d["bodies"]["HP"]["position_mm"][7], [7, 14, 600])
    assert np.all(np.isnan(d["bodies"]["CT"]["pose_4x4"]))
    assert np.allclose(d["quality"]["MP"]["fit_rms_mm"], 7.0)
    m = d["markers_3d"]
    assert len(m["frame_row"]) == 36
    assert 4 not in set(m["frame_row"].tolist())
    sel = m["frame_row"] == 7
    assert np.allclose(m["position_mm"][sel][:, 0], 7.0)
    assert list(m["corner_index"][sel]) == [0, 1, 2, 3]
    with h5py.File(path, "r") as f:
        assert f["markers_3d"].attrs["layout"] == "flat_v1"
        assert not any(isinstance(v, h5py.Group) for v in f["markers_3d"].values())


def test_reads_old_nested_markers_layout(tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("markers_3d")
        for row in (0, 2):
            fg = g.create_group(f"frame_{row:06d}")
            for j in range(3):
                pg = fg.create_group(f"point_{j:03d}")
                pg.attrs["marker_id"] = 20 + j
                pg.attrs["corner_index"] = j
                pg.create_dataset("position_mm", data=np.array([row, j, 1.0]))
    m = read_markers_3d(path)
    assert list(m["frame_row"]) == [0, 0, 0, 2, 2, 2]
    assert list(m["marker_id"]) == [20, 21, 22] * 2
    assert np.allclose(m["position_mm"][4], [2, 1, 1])


def test_zero_frames_writes_valid_empty_session(tmp_path):
    path = write_tracking_session(tmp_path / "empty.h5", frames=[], notes="aborted")
    d = read_tracking_session(path)
    assert d["meta"]["n_frames"] == 0
    assert d["bodies"]["HP"]["pose_4x4"].shape == (0, 4, 4)
    assert "markers_3d" not in d


def test_incremental_writer_matches_and_survives_partial(tmp_path):
    frames = [_frame(i, cams=("cam0",) if i < 5 else ("cam0", "cam1")) for i in range(23)]
    w = TrackingSessionWriter(tmp_path / "inc.h5", method="pnp", flush_every=4)
    for fr in frames[:13]:
        w.append(fr)
    # Flushed rows are already on disk before close: a copy taken now (what a
    # crash would leave behind) is a readable file with 12 rows.
    shutil.copy(tmp_path / "inc.h5", tmp_path / "crash.h5")
    crashed = read_tracking_session(tmp_path / "crash.h5")
    assert crashed["meta"]["n_frames"] == 12
    assert crashed["bodies"]["HP"]["pose_4x4"].shape == (12, 4, 4)
    w.extend(frames[13:])
    w.set_meta(pair_mismatch=0)
    w.close(notes="done")
    d = read_tracking_session(tmp_path / "inc.h5")
    assert d["meta"]["n_frames"] == 23 and d["meta"]["notes"] == "done"
    cam1 = d["timestamps"]["t_cam_ns"]["cam1"]
    assert np.all(np.isnan(cam1[:5])) and np.all(np.isfinite(cam1[5:]))
    one = read_tracking_session(write_tracking_session(tmp_path / "one.h5", frames=frames))
    for body in ("HP", "MP", "CT"):
        assert np.array_equal(
            one["bodies"][body]["pose_4x4"], d["bodies"][body]["pose_4x4"], equal_nan=True
        )
    assert np.array_equal(one["markers_3d"]["frame_row"], d["markers_3d"]["frame_row"])


def test_flat_markers_stay_small(tmp_path):
    frames = []
    for i in range(2000):
        fr = FrameRecord(frame_idx=i, t_master_us=i * 33_333)
        fr.markers_3d = [
            {"marker_id": j // 4, "corner_index": j % 4, "position_mm": [0.0, 0.0, 0.0]}
            for j in range(24)
        ]
        frames.append(fr)
    path = write_tracking_session(tmp_path / "big.h5", frames=frames, method="triangulation")
    # 48k points as one group + dataset each ran to tens of MB (~1.4 GB per
    # 20 min take); flat arrays are ~36 B/point plus the pose tables.
    assert path.stat().st_size < 5_000_000


def test_failed_flush_rolls_back_and_keeps_the_frames(tmp_path, monkeypatch):
    """Disk full / Ctrl+C mid-flush must not leave timestamps longer than bodies."""
    w = TrackingSessionWriter(tmp_path / "s.h5", flush_every=1000)
    w.append(_frame(0))
    w.flush()
    w.append(_frame(1))
    w.append(_frame(2))
    real = TrackingSessionWriter._append
    calls = {"n": 0}

    def flaky(self, ds, data):
        calls["n"] += 1
        if calls["n"] == 5:
            raise OSError("No space left on device")
        real(self, ds, data)

    monkeypatch.setattr(TrackingSessionWriter, "_append", flaky)
    with pytest.raises(OSError):
        w.flush()
    monkeypatch.setattr(TrackingSessionWriter, "_append", real)
    with h5py.File(w.path, "r") as f:  # rolled back to the one flushed frame everywhere
        assert f["timestamps"]["frame_idx"].shape[0] == 1
        assert f["bodies"]["HP"]["pose_4x4"].shape[0] == 1
    w.close()  # the kept frames are written on the next flush
    d = read_tracking_session(w.path)
    assert list(d["timestamps"]["frame_idx"]) == [0, 1, 2]
    assert d["meta"]["n_frames"] == 3
