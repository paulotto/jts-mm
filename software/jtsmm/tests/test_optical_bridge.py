# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""OpticalTrackBridge stereo pairing: wait for both cameras, never triangulate 1-cam."""

from __future__ import annotations

import queue
from collections import Counter
from types import SimpleNamespace

from jtsmm.capture.optical_bridge import OpticalTrackBridge, track_stride

_PERIOD_NS = 16_667_000  # 60 Hz


def _bare_bridge(
    *,
    cam_ids=("cam0", "cam1"),
    queue_size: int = 8,
    pending_max: int = 120,
    pending_hold_ns: int = 1_000_000_000,
    match_ns: int = 8_333_000,
):
    bridge = OpticalTrackBridge.__new__(OpticalTrackBridge)
    bridge.cam_ids = set(cam_ids)
    bridge._profile_cam_ids = list(cam_ids)
    bridge.stats = Counter()
    bridge._pending = {cid: [] for cid in cam_ids}
    bridge._newest_ns = {}
    bridge._frame_idx = 0
    bridge._jobs = queue.Queue(maxsize=queue_size)
    bridge._results = queue.Queue()
    bridge._stopped = False
    bridge._PENDING_MAX = pending_max
    bridge._PENDING_HOLD_NS = pending_hold_ns
    bridge._MATCH_NS = match_ns
    return bridge


def _drain_jobs(bridge: OpticalTrackBridge) -> list[tuple]:
    out = []
    while True:
        try:
            job = bridge._jobs.get_nowait()
        except queue.Empty:
            return out
        out.append(job)


def _pending_n(bridge: OpticalTrackBridge) -> int:
    return sum(len(q) for q in bridge._pending.values())


def test_calibration_ignores_the_configured_track_rate():
    from jtsmm.config.registry import effective_optical_track_hz

    cfg = {"session": {"optical_track_hz": 10}}
    assert effective_optical_track_hz(cfg) == 10
    assert effective_optical_track_hz(cfg, ct_calibration=True) is None
    assert effective_optical_track_hz(cfg, protocol="imu_mounting") is None


def test_track_stride_from_hz():
    assert track_stride(None, 30) == 1
    assert track_stride(30, 30) == 1
    assert track_stride(40, 30) == 1
    assert track_stride(15, 30) == 2
    assert track_stride(10, 30) == 3
    assert track_stride(12, 30) == 3  # halfway rounds to the slower rate
    assert track_stride(20, 30) == 2


def test_track_stride_enqueues_every_nth_pair_and_does_not_count_a_drop():
    bridge = _bare_bridge()
    bridge._track_stride = 3
    bridge._pairs_seen = 0
    for i in range(6):
        t_ns = i * 20_000_000
        bridge.note_rgb(cam_id="cam0", t_master_us=i, rgb=f"a{i}", sensor_timestamp_ns=t_ns)
        bridge.note_rgb(
            cam_id="cam1", t_master_us=i, rgb=f"b{i}", sensor_timestamp_ns=t_ns + 1000
        )
    jobs = _drain_jobs(bridge)
    assert [job[1] for job in jobs] == [0, 3]
    assert bridge.stats["paired"] == 6
    assert bridge.stats["enqueued"] == 2
    assert bridge.stats["track_skipped"] == 4
    assert bridge.stats["pair_dropped"] == 0
    assert bridge.stats["queue_dropped"] == 0


def test_note_rgb_holds_incomplete_until_pair_arrives():
    bridge = _bare_bridge()
    rgb0 = SimpleNamespace(cam="cam0")
    rgb1 = SimpleNamespace(cam="cam1")

    bridge.note_rgb(cam_id="cam0", t_master_us=1000, rgb=rgb0, sensor_timestamp_ns=10)
    assert bridge._jobs.empty()
    assert _pending_n(bridge) == 1
    assert bridge.stats["paired"] == 0

    bridge.flush()
    assert bridge._jobs.empty()
    assert _pending_n(bridge) == 1

    bridge.note_rgb(cam_id="cam1", t_master_us=1000, rgb=rgb1, sensor_timestamp_ns=11)
    jobs = _drain_jobs(bridge)
    assert len(jobs) == 1
    _fidx, t_us, frames, t_cam = jobs[0]
    assert t_us == 1000
    assert set(frames) == {"cam0", "cam1"}
    assert frames["cam0"] is rgb0
    assert frames["cam1"] is rgb1
    assert t_cam == {"cam0": 10, "cam1": 11}
    assert _pending_n(bridge) == 0
    assert bridge.stats["paired"] == 1
    assert bridge.stats["pair_dropped"] == 0


def test_flush_does_not_enqueue_stale_one_cam_bundles():
    """Regression: old flush() triangulated incompletes once >4 timestamps piled up."""
    bridge = _bare_bridge()
    for i in range(8):
        bridge.note_rgb(
            cam_id="cam0",
            t_master_us=1000 + i,
            rgb=f"cam0-{i}",
            sensor_timestamp_ns=i * _PERIOD_NS,
        )
    bridge.flush()
    assert bridge._jobs.empty()
    assert _pending_n(bridge) == 8
    assert bridge.stats["paired"] == 0
    assert bridge.stats["pair_dropped"] == 0


def test_late_second_camera_still_pairs_after_several_flush_calls():
    bridge = _bare_bridge()
    bridge.note_rgb(cam_id="cam0", t_master_us=5000, rgb="a", sensor_timestamp_ns=1)
    for _ in range(6):
        bridge.flush()
    assert bridge._jobs.empty()

    bridge.note_rgb(cam_id="cam1", t_master_us=5000, rgb="b", sensor_timestamp_ns=2)
    jobs = _drain_jobs(bridge)
    assert len(jobs) == 1
    assert set(jobs[0][2]) == {"cam0", "cam1"}


def test_note_rgb_does_not_drop_before_second_camera_in_batch():
    """cam0 can run ahead by >60 frames; cam1 must still pair the first shutter."""
    bridge = _bare_bridge(pending_max=200, pending_hold_ns=3_000_000_000)
    n = 80
    t0 = 1_000_000
    for i in range(n):
        bridge.note_rgb(
            cam_id="cam0",
            t_master_us=t0 + i,
            rgb=f"cam0-{i}",
            sensor_timestamp_ns=i * _PERIOD_NS,
        )
    assert bridge.stats["pair_dropped"] == 0
    bridge.note_rgb(
        cam_id="cam1",
        t_master_us=t0,
        rgb="cam1-0",
        sensor_timestamp_ns=0,
    )
    jobs = _drain_jobs(bridge)
    assert len(jobs) == 1
    assert jobs[0][1] == t0
    assert set(jobs[0][2]) == {"cam0", "cam1"}


def test_flush_drops_when_other_camera_has_passed_the_shutter():
    """cam1's later pulse means cam0's unmatched early frame is gone, not a pair."""
    bridge = _bare_bridge()
    bridge.note_rgb(cam_id="cam0", t_master_us=0, rgb="old", sensor_timestamp_ns=0)
    bridge.note_rgb(
        cam_id="cam1",
        t_master_us=2_000_000,
        rgb="late",
        sensor_timestamp_ns=2_000_000_000,
    )
    bridge.flush()
    assert bridge.stats["pair_dropped"] == 1
    assert len(bridge._pending["cam0"]) == 0
    assert len(bridge._pending["cam1"]) == 1
    assert bridge._jobs.empty()


def test_shutdown_flush_drops_incompletes_not_triangulated():
    bridge = _bare_bridge()
    bridge.note_rgb(cam_id="cam0", t_master_us=1, rgb="x", sensor_timestamp_ns=1)
    bridge.flush(force=True)
    assert bridge._jobs.empty()
    assert _pending_n(bridge) == 0
    assert bridge.stats["pair_dropped"] == 1
    assert bridge.stats["paired"] == 0


def test_single_camera_profile_enqueues_immediately():
    bridge = _bare_bridge(cam_ids=("cam0",))
    bridge.note_rgb(cam_id="cam0", t_master_us=9, rgb="solo", sensor_timestamp_ns=1)
    jobs = _drain_jobs(bridge)
    assert len(jobs) == 1
    assert list(jobs[0][2]) == ["cam0"]
    assert bridge.stats["paired"] == 1


def test_pairs_by_sensor_time_not_trigger_id():
    """Same shutter, disagreeing MCU ids: still a stereo pair."""
    bridge = _bare_bridge()
    bridge.note_rgb(cam_id="cam0", t_master_us=100, rgb="a", sensor_timestamp_ns=50_000_000)
    bridge.note_rgb(cam_id="cam1", t_master_us=999_999, rgb="b", sensor_timestamp_ns=50_100_000)
    jobs = _drain_jobs(bridge)
    assert len(jobs) == 1
    assert jobs[0][1] == 100  # reference camera stamp
    assert jobs[0][2] == {"cam0": "a", "cam1": "b"}
    assert bridge.stats["paired"] == 1


def test_does_not_pair_same_trigger_id_far_apart_in_time():
    """The session bug: same guessed id, 3.35 s apart in SensorTimestamp."""
    bridge = _bare_bridge()
    bridge.note_rgb(
        cam_id="cam0",
        t_master_us=1401,
        rgb="early",
        sensor_timestamp_ns=1_000_000_000,
    )
    bridge.note_rgb(
        cam_id="cam1",
        t_master_us=1401,
        rgb="late",
        sensor_timestamp_ns=1_000_000_000 + 3_350_000_000,
    )
    jobs = _drain_jobs(bridge)
    assert jobs == []
    assert bridge.stats["paired"] == 0
    bridge.flush()
    assert bridge.stats["paired"] == 0
    assert bridge.stats["pair_dropped"] == 1


def test_skips_dropped_pulse_then_pairs_rest():
    """cam1 missed the first shutter; remaining pulses still pair in order."""
    bridge = _bare_bridge()
    for i in range(3):
        bridge.note_rgb(
            cam_id="cam0",
            t_master_us=10 + i,
            rgb=f"c0-{i}",
            sensor_timestamp_ns=i * _PERIOD_NS,
        )
    for i in (1, 2):
        bridge.note_rgb(
            cam_id="cam1",
            t_master_us=90 + i,
            rgb=f"c1-{i}",
            sensor_timestamp_ns=i * _PERIOD_NS,
        )
    jobs = _drain_jobs(bridge)
    assert len(jobs) == 2
    assert [j[1] for j in jobs] == [11, 12]
    assert jobs[0][2]["cam1"] == "c1-1"
    assert jobs[1][2]["cam1"] == "c1-2"
    assert len(bridge._pending["cam0"]) == 0  # missed first shutter dropped as stale
    assert bridge.stats["paired"] == 2
    assert bridge.stats["pair_dropped"] == 1
