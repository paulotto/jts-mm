# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Replay uses recorded timestamps, the recording's camera set, one solve per frame.

Pure numpy + h5py: frame loading and the tracker are stubbed, so no cv2.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import optical.tracking.replay as replay
from optical.tracking.frame_store import (
    TIMESTAMPS_NAME,
    FrameTimestampLog,
    load_frame_timestamps,
    write_session_meta,
)
from optical.tracking.io import FrameRecord, read_tracking_session
from optical.tracking.pose import nan_pose
from optical.tracking.pose_estimation import WorldBodyPose
from optical.tracking.tracker import LastEstimate


def test_timestamp_log_round_trip_and_torn_line(tmp_path):
    with FrameTimestampLog(tmp_path) as log:
        log.append(0, t_cam_ns={"cam0": 10, "cam1": 11}, t_master_us=100)
        log.append(1, t_cam_ns={"cam0": 20}, t_master_us=None)
    with (tmp_path / TIMESTAMPS_NAME).open("a") as fh:
        fh.write('{"frame_idx": 2, "t_ca')  # process killed mid-line
    stamps = load_frame_timestamps(tmp_path)
    assert stamps == {0: ({"cam0": 10, "cam1": 11}, 100), 1: ({"cam0": 20}, None)}
    assert load_frame_timestamps(tmp_path / "nope") == {}


class _StubTracker:
    def __init__(self):
        self.config = SimpleNamespace(pose_method="pnp", triangulation_min_views=2,
                                      marker_config_path="x", calibration_bundle_path="y")
        self.rig_id = "rig"
        self.calls = 0
        self.last_detections = {}
        self.last_estimate = None

    def process_frame(self, frames, *, frame_idx, t_cam_ns=None, t_master_us=None):
        self.calls += 1
        poses = {b: WorldBodyPose(b, nan_pose(), 0, 0, float("nan"), False) for b in ("HP", "MP", "CT")}
        self.last_estimate = LastEstimate("pnp", poses, None, {})
        return FrameRecord(frame_idx=frame_idx, t_cam_ns=dict(t_cam_ns or {}), t_master_us=t_master_us)


@pytest.fixture
def dump(tmp_path, monkeypatch):
    frames_dir = tmp_path / "leaf"
    write_session_meta(frames_dir, cam_ids=["cam0", "cam1"], frame_rate_hz=30.0,
                       image_size=(728, 544))
    fake = {i: {"cam0": np.zeros((4, 4), np.uint8), "cam1": np.zeros((4, 4), np.uint8)}
            for i in range(6)}
    for i in fake:
        (frames_dir / "frames" / f"{i:06d}").mkdir(parents=True)
    monkeypatch.setattr(replay, "iter_frames", lambda d: iter(sorted(fake.items())))
    stub = _StubTracker()
    monkeypatch.setattr(replay, "load_tracker_for_recording", lambda cfg, d: (stub, stub.config))
    return frames_dir, stub


def _run(frames_dir, out):
    replay.run_replay(tracker_config_path=frames_dir / "optical.yaml", frames_dir=frames_dir,
                      output_path=out, experiment_name="t", print_every=0, verbose=False,
                      max_frames=None, debug_only=False)
    return read_tracking_session(out)


def test_replay_writes_recorded_timestamps(dump, tmp_path):
    frames_dir, stub = dump
    with FrameTimestampLog(frames_dir) as log:
        for i in range(6):
            log.append(i, t_cam_ns={"cam0": 7_000_000_000 + i * 33_333_333,
                                    "cam1": 7_000_000_050 + i * 33_333_333},
                       t_master_us=900_000 + i * 33_333)
    d = _run(frames_dir, tmp_path / "out.h5")
    assert stub.calls == 6  # one detect + solve per frame, not two
    assert list(d["timestamps"]["t_master_us"]) == [900_000 + i * 33_333 for i in range(6)]
    assert d["timestamps"]["t_cam_ns"]["cam1"][2] == 7_000_000_050 + 2 * 33_333_333
    assert d["meta"]["frame_rate_hz"] == pytest.approx(30.0)
    assert d["meta"]["replay_timestamps"] == "recorded"


def test_replay_without_stamps_synthesises_from_rate(dump, tmp_path):
    frames_dir, _stub = dump
    d = _run(frames_dir, tmp_path / "out.h5")
    t0 = d["timestamps"]["t_cam_ns"]["cam0"]
    assert np.allclose(np.diff(t0), 1e9 / 30.0)
    assert np.all(np.isnan(d["timestamps"]["t_master_us"]))
    assert d["meta"]["replay_timestamps"] == "synthesised"


def test_replay_profile_follows_recorded_cameras(tmp_path, monkeypatch):
    from optical.tracking import tracker as trk

    write_session_meta(tmp_path, cam_ids=["cam0", "cam1"], image_size=(728, 544))
    seen = {}

    def fake_from_path(path, *, active_cam_ids=None, n_cameras=None):
        seen["active_cam_ids"] = active_cam_ids
        raise RuntimeError("stop")

    monkeypatch.setattr(trk.TrackerConfig, "from_path", staticmethod(fake_from_path))
    with pytest.raises(RuntimeError, match="stop"):
        trk.load_tracker_for_recording(tmp_path / "optical.yaml", tmp_path)
    assert seen["active_cam_ids"] == ["cam0", "cam1"]


def test_replay_keeps_rows_on_live_frame_idx_when_the_dump_dropped_frames(tmp_path, monkeypatch):
    """The async dump may drop PNGs; rows must still equal the live frame_idx
    (the live take's JTS intervals index those rows)."""
    frames_dir = tmp_path / "leaf"
    write_session_meta(frames_dir, cam_ids=["cam0"], frame_rate_hz=30.0, image_size=(728, 544))
    stored = [0, 1, 4, 5]  # frames 2 and 3 never reached the SD card
    fake = {i: {"cam0": np.zeros((4, 4), np.uint8)} for i in stored}
    for i in stored:
        (frames_dir / "frames" / f"{i:06d}").mkdir(parents=True)
    monkeypatch.setattr(replay, "iter_frames", lambda d: iter(sorted(fake.items())))
    stub = _StubTracker()
    monkeypatch.setattr(replay, "load_tracker_for_recording", lambda cfg, d: (stub, stub.config))
    with FrameTimestampLog(frames_dir) as log:
        for i in range(6):
            log.append(i, t_cam_ns={"cam0": 7_000_000_000 + i * 33_333_333}, t_master_us=900_000 + i * 33_333)
    d = _run(frames_dir, tmp_path / "out.h5")
    assert stub.calls == 4
    assert list(d["timestamps"]["frame_idx"]) == [0, 1, 2, 3, 4, 5]
    assert list(d["timestamps"]["t_master_us"]) == [900_000 + i * 33_333 for i in range(6)]
