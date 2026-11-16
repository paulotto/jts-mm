# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Same-trigger pairing by SensorTimestamp; each captured frame evaluated at most once."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from _fix_J2_imports import import_optical

(metrics,) = import_optical("optical.evaluation.metrics")


def test_detection_rate_uses_evaluated_frames() -> None:
    res = metrics.aggregate_marker_results(
        marker_id=1, size_mm=30.0, label="30mm", total_frames=300,
        detected_flags=[True, True, True, False], samples=[], thresholds={},
        tri_detected_flags=[True, False], tri_samples=[], tri_thresholds={},
        evaluate_jitter=False,
    )
    assert res.detection_rate_pct == pytest.approx(75.0)
    assert res.tri_detection_rate_pct == pytest.approx(50.0)


class _ScriptedRig:
    """CameraRig whose cameras deliver frames with scripted sensor timestamps (ns)."""

    def __new__(cls, ts_by_cam):
        from optical.calibration.multicam import CameraRig
        from optical.sync.frame_meta import CamFrameMeta

        class Rig(CameraRig):
            def __init__(self):
                self._it = {cid: iter(ts) for cid, ts in ts_by_cam.items()}
                self.reads = {cid: 0 for cid in ts_by_cam}

            def capture_all(self):
                return self.capture_all_with_meta()[0]

            def _capture_one(self, cid):
                t = next(self._it[cid], None)
                if t is None:
                    return None
                self.reads[cid] += 1
                return np.full((2, 2), t % 255, np.uint8), CamFrameMeta(cid, t, t)

            def capture_all_with_meta(self):
                frames, meta = {}, {}
                for cid in self._it:
                    got = self._capture_one(cid)
                    if got is not None:
                        frames[cid], meta[cid] = got
                return frames, meta

            def close(self):
                pass

        return Rig()


P = 16_666_667  # 60 Hz trigger period


def test_capture_all_synced_realigns_a_lagging_camera() -> None:
    pytest.importorskip("cv2")
    # cam1 still has a stale frame queued: it returns trigger k-1 while cam0 returns k.
    rig = _ScriptedRig({"cam0": [10 * P, 11 * P], "cam1": [9 * P + 300_000, 10 * P + 200_000]})
    frames, meta = rig.capture_all_synced()
    assert set(frames) == {"cam0", "cam1"}
    assert abs(meta["cam0"].sensor_timestamp_ns - meta["cam1"].sensor_timestamp_ns) < 1_000_000
    assert rig.reads == {"cam0": 1, "cam1": 2}
    assert rig.sync_stats["resynced"] == 1


def test_picam_sync_holds_newer_request_and_copies_pixels_after_match() -> None:
    """A full-frame copy must not sit between the two capture_request calls.

    At 30 Hz the cameras are one period (33.3 ms) apart when the second read
    happens after the first frame was copied. Holding the newer request and
    only re-reading the laggard closes that gap before any pixel copy.
    """
    pytest.importorskip("cv2")
    from optical.calibration.multicam import CamSpec, PicamRig

    period = 33_333_333
    script = {
        "cam0": [10 * period],
        "cam1": [9 * period, 10 * period + 100_000],
    }
    events: list[tuple] = []

    class _Req:
        def __init__(self, ts: int) -> None:
            self.ts = ts

        def get_metadata(self):
            return {"SensorTimestamp": self.ts}

        def release(self) -> None:
            events.append(("release", self.ts))

        def make_array(self, _name: str):
            events.append(("copy", self.ts))
            return np.zeros((4, 4, 3), np.uint8)

    class _Cam:
        def __init__(self, cid: str) -> None:
            self._ts = iter(script[cid])
            self.cid = cid

        def capture_request(self) -> _Req:
            ts = next(self._ts)
            events.append(("grab", self.cid, ts))
            return _Req(ts)

    rig = PicamRig.__new__(PicamRig)
    rig._cams = {
        "cam0": CamSpec("cam0", 0, (4, 4)),
        "cam1": CamSpec("cam1", 1, (4, 4)),
    }
    rig._frame_idx = {"cam0": 0, "cam1": 0}
    rig._picams = {"cam0": _Cam("cam0"), "cam1": _Cam("cam1")}

    frames, meta = rig.capture_all_synced()
    assert set(frames) == {"cam0", "cam1"}
    assert abs(meta["cam0"].sensor_timestamp_ns - meta["cam1"].sensor_timestamp_ns) < 1_000_000
    grabs = [e for e in events if e[0] == "grab"]
    assert grabs == [
        ("grab", "cam0", 10 * period),
        ("grab", "cam1", 9 * period),
        ("grab", "cam1", 10 * period + 100_000),
    ]
    first_copy = next(i for i, e in enumerate(events) if e[0] == "copy")
    assert all(e[0] != "copy" for e in events[:first_copy])
    assert ("release", 10 * period) not in events[:first_copy]


def test_capture_all_synced_gives_up_instead_of_mispairing() -> None:
    pytest.importorskip("cv2")
    # cam1 permanently half a period off (e.g. untriggered): never a valid pair.
    rig = _ScriptedRig({"cam0": [k * P for k in range(10)], "cam1": [k * P + P // 2 for k in range(10)]})
    frames, meta = rig.capture_all_synced(max_retries=2)
    assert frames == {} and meta == {}
    assert rig.sync_stats["dropped"] == 1


def test_live_feed_hands_out_each_frame_set_once() -> None:
    pytest.importorskip("cv2")
    from optical.evaluation.run_marker_size_eval import LivePreviewFeed

    feed = LivePreviewFeed(None, [], ["cam0"], detector=None, markers=[], preview=None)
    with feed._new_frame:
        feed._latest = {"cam0": np.zeros((2, 2), np.uint8)}
        feed._seq = 1
    seq, frames = feed.latest_frames_seq(-1, timeout_s=0.1)
    assert seq == 1 and "cam0" in frames
    # Same set again -> times out empty instead of returning a duplicate.
    seq2, frames2 = feed.latest_frames_seq(seq, timeout_s=0.05)
    assert seq2 == 1 and frames2 == {}

    def bump():
        with feed._new_frame:
            feed._seq = 2
            feed._new_frame.notify_all()

    threading.Timer(0.02, bump).start()
    seq3, frames3 = feed.latest_frames_seq(seq, timeout_s=1.0)
    assert seq3 == 2 and frames3
